//! The walk on near-degenerate input (decision N5): ring vertices on, or
//! within rounding of, the mesh's corners, sides, spokes and split
//! centres; segments through them and along sides; shallow crossings;
//! long segments past an outer corner.
//!
//! Three oracles, none of them the walk:
//!
//! - the world clip: each cell's four split triangles clipped against the
//!   ring in world space, as a fraction of the cell's area (computed from
//!   the triangles, not `cell_area`'s shoelace over absolute coordinates);
//! - complementary tilings: pieces that tile a polygon (cut along chords
//!   through the degenerate points, or fanned round one) cover each cell
//!   as the polygon does;
//! - the same ring reversed (its index ring the same reversed, bit for
//!   bit) and started at another vertex.
//!
//! For a longer search, `WALK_CASES=20 cargo test --release --test
//! mesh_walk_degenerate` runs twenty times the cases, and `WALK_SHOW=name`
//! prints the failures whose case name starts with `name` (`WALK_SHOW=`
//! for all).

use controlledburn::CoverageWindow;
use rwarp::mesh::{CellMap, QuadMesh};

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.u() * n as f64) as usize).min(n - 1)
    }
    fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[self.below(v.len())]
    }
}

type Pt = [f64; 2];

fn add(a: Pt, b: Pt) -> Pt {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}
fn mul(a: Pt, s: f64) -> Pt {
    [a[0] * s, a[1] * s]
}
fn lerp(a: Pt, b: Pt, t: f64) -> Pt {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}
fn cross(a: Pt, b: Pt) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn dot(a: Pt, b: Pt) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn norm(a: Pt) -> f64 {
    a[0].hypot(a[1])
}

// ---- meshes ----------------------------------------------------------------

fn lattice(ncol: usize, nrow: usize, f: impl Fn(f64, f64) -> Pt) -> QuadMesh {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=nrow {
        for i in 0..=ncol {
            let p = f(i as f64, j as f64);
            x.push(p[0]);
            y.push(p[1]);
        }
    }
    QuadMesh::from_corners(ncol, nrow, x, y, CellMap::EqualArea).unwrap()
}

fn jittered(seed: u64, amount: f64) -> QuadMesh {
    let mut g = Lcg(seed);
    let (nc, nr) = (9, 7);
    let jit: Vec<Pt> = (0..(nc + 1) * (nr + 1))
        .map(|_| [(g.u() - 0.5) * 2.0 * amount, (g.u() - 0.5) * 2.0 * amount])
        .collect();
    lattice(nc, nr, |i, j| {
        let k = j as usize * (nc + 1) + i as usize;
        [0.7 * i + jit[k][0] * 0.7, 3.0 - 0.5 * j + jit[k][1] * 0.5]
    })
}

/// Cells with dents: three corners pushed in, so their cells are not
/// convex and the split centre is not the corner mean.
fn dented() -> QuadMesh {
    let (nc, nr) = (6, 5);
    let dent = [(1usize, 1usize, 0.7), (2, 4, 0.6), (4, 2, -0.65)];
    lattice(nc, nr, |i, j| {
        let d = dent
            .iter()
            .find(|&&(dj, di, _)| dj as f64 == j && di as f64 == i)
            .map_or(0.0, |t| t.2);
        [i + d, j + d]
    })
}

/// A rotated regular grid: its sides lie on lines, which the world points
/// are only within rounding of.
fn rotated(d: f64, a: f64, o: Pt) -> QuadMesh {
    let (ca, sa) = (a.cos(), a.sin());
    lattice(10, 8, |i, j| [o[0] + d * (i * ca - j * sa), o[1] + d * (i * sa + j * ca)])
}

fn meshes() -> Vec<(&'static str, QuadMesh)> {
    vec![
        // integer corners: the orientation tests are exact
        ("square", lattice(8, 6, |i, j| [i, j])),
        ("rotated", rotated(1.7, 0.52, [3.0, -2.0])),
        ("rotated_far", rotated(0.37, 2.2, [1.0e3, -2.0e3])),
        ("rotated_small", rotated(1.3e-3, -0.9, [0.4, 0.1])),
        ("jitter", jittered(7, 0.2)),
        ("jitter_strong", jittered(11, 0.45)),
        (
            "polar",
            lattice(12, 6, |i, j| {
                let (r, a) = (100.0 + 20.0 * j, (15.0 * i).to_radians());
                [r * a.cos(), r * a.sin()]
            }),
        ),
        (
            "sinusoid",
            lattice(10, 8, |i, j| [i + 0.3 * (j * 0.7).sin(), j + 0.25 * (i * 0.9).cos() + 0.1 * i]),
        ),
        ("dented", dented()),
        // stored bottom row first, so the cells turn the other way
        ("flipped", lattice(8, 6, |i, j| [0.9 * i + 0.05 * j, 6.0 - 1.1 * j + 0.03 * i])),
    ]
}

/// The mesh's typical cell size.
fn scale(m: &QuadMesh) -> f64 {
    let n = (m.ncol() * m.nrow()) as f64;
    let a: f64 = (0..m.nrow())
        .flat_map(|r| (0..m.ncol()).map(move |c| (r, c)))
        .map(|(r, c)| m.cell_area(r, c))
        .sum();
    (a / n).sqrt()
}

// ---- degenerate points ------------------------------------------------------

/// A point of the mesh's structure: a corner, a split centre, a point on
/// a side or on a spoke (within rounding of it, as `lerp` places it).
#[derive(Clone, Copy, Debug)]
struct Feature {
    p: Pt,
    /// A direction along the feature (a side's or a spoke's), or any
    /// side's at a corner or a centre.
    dir: Pt,
}

fn features(m: &QuadMesh, g: &mut Lcg) -> Vec<Feature> {
    let mut out = Vec::new();
    let ts = [0.5, 1e-9, 1.0 - 1e-9, 1e-14];
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let q = m.corners(r, c);
            let x = m.split_centre(r, c).0;
            out.push(Feature { p: q[0], dir: sub(q[1], q[0]) });
            out.push(Feature { p: q[0], dir: sub(q[3], q[0]) });
            out.push(Feature { p: x, dir: sub(q[g.below(4)], x) });
            for k in 0..4 {
                let (a, b) = (q[k], q[(k + 1) % 4]);
                let t = if g.u() < 0.5 { g.u() } else { g.pick(&ts) };
                out.push(Feature { p: lerp(a, b, t), dir: sub(b, a) });
                let t = if g.u() < 0.5 { g.u() } else { g.pick(&ts) };
                out.push(Feature { p: lerp(x, a, t), dir: sub(a, x) });
            }
        }
        // the last column's and the last row's corners
        let q = m.corners(r, m.ncol() - 1);
        out.push(Feature { p: q[1], dir: sub(q[2], q[1]) });
    }
    for c in 0..m.ncol() {
        let q = m.corners(m.nrow() - 1, c);
        out.push(Feature { p: q[3], dir: sub(q[2], q[3]) });
        out.push(Feature { p: q[2], dir: sub(q[2], q[3]) });
    }
    out
}

/// `p`, or `p` moved by a distance between one rounding unit and 1e-10 of
/// the cell size.
fn jiggle(g: &mut Lcg, p: Pt, h: f64) -> Pt {
    if g.u() < 0.4 {
        return p;
    }
    let d = h * g.pick(&[f64::EPSILON, 1e-15, 1e-14, 1e-13, 1e-12, 1e-10]);
    let a = g.u() * std::f64::consts::TAU;
    add(p, [d * a.cos(), d * a.sin()])
}

// ---- polygons ---------------------------------------------------------------

fn shoelace(p: &[Pt]) -> f64 {
    let n = p.len();
    let o = p[0];
    (1..n.saturating_sub(1))
        .map(|k| cross(sub(p[k], o), sub(p[k + 1], o)))
        .sum::<f64>()
        * 0.5
}

/// Sutherland-Hodgman: ring `p` clipped by convex counter-clockwise `clip`.
fn clip(p: &[Pt], clip: &[Pt]) -> Vec<Pt> {
    let mut out = p.to_vec();
    let n = clip.len();
    for k in 0..n {
        let (a, b) = (clip[k], clip[(k + 1) % n]);
        let side = |q: Pt| cross(sub(b, a), sub(q, a));
        let input = std::mem::take(&mut out);
        let m = input.len();
        for i in 0..m {
            let (s, e) = (input[(i + m - 1) % m], input[i]);
            let (ss, se) = (side(s), side(e));
            if se >= 0.0 {
                if ss < 0.0 {
                    out.push(lerp(s, e, ss / (ss - se)));
                }
                out.push(e);
            } else if ss >= 0.0 {
                out.push(lerp(s, e, ss / (ss - se)));
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// Convex polygon `poly` cut by the line through `d` along `v`: the parts
/// either side, each with `d` as a vertex where it lies on the cut. A cut
/// point on an edge is computed from the edge's ends in a fixed order, so
/// the two parts share it bitwise.
fn cut(poly: &[Pt], d: Pt, v: Pt) -> Vec<Vec<Pt>> {
    let n = poly.len();
    let side: Vec<f64> = poly.iter().map(|&q| cross(v, sub(q, d))).collect();
    let mut parts = [Vec::new(), Vec::new()];
    let mut on: [Vec<bool>; 2] = [Vec::new(), Vec::new()];
    for i in 0..n {
        let (s, e) = (poly[i], poly[(i + 1) % n]);
        let (ss, se) = (side[i], side[(i + 1) % n]);
        if ss >= 0.0 {
            parts[0].push(s);
            on[0].push(ss == 0.0);
        }
        if ss <= 0.0 {
            parts[1].push(s);
            on[1].push(ss == 0.0);
        }
        if (ss > 0.0 && se < 0.0) || (ss < 0.0 && se > 0.0) {
            let x = if (s[0], s[1]) < (e[0], e[1]) {
                lerp(s, e, ss / (ss - se))
            } else {
                lerp(e, s, se / (se - ss))
            };
            for k in 0..2 {
                parts[k].push(x);
                on[k].push(true);
            }
        }
    }
    let mut out = Vec::new();
    for k in 0..2 {
        let (mut p, o) = (std::mem::take(&mut parts[k]), &on[k]);
        if p.len() < 3 {
            continue;
        }
        // `d` between the two cut points, if it is
        let m = p.len();
        if let Some(i) = (0..m).find(|&i| o[i] && o[(i + 1) % m]) {
            let (a, b) = (p[i], p[(i + 1) % m]);
            let (ta, tb) = (dot(sub(a, d), v), dot(sub(b, d), v));
            if ta * tb < 0.0 {
                p.insert(i + 1, d);
            }
        }
        if shoelace(&p).abs() > 0.0 {
            out.push(p);
        }
    }
    out
}

/// The convex quadrilateral round `c`: a rectangle of half sizes `w`, `h`
/// turned by `a`.
fn quad(c: Pt, w: f64, h: f64, a: f64) -> Vec<Pt> {
    let (ca, sa) = (a.cos(), a.sin());
    [[-w, -h], [w, -h], [w, h], [-w, h]]
        .iter()
        .map(|&[x, y]| [c[0] + x * ca - y * sa, c[1] + x * sa + y * ca])
        .collect()
}

// ---- coverage ---------------------------------------------------------------

/// The walk's coverage of each cell (f64, through `CoverageWindow`).
fn cover(m: &QuadMesh, ring: &[Pt]) -> Result<Vec<f64>, String> {
    let mut win = CoverageWindow::default();
    let mut out = vec![0.0; m.ncol() * m.nrow()];
    if m.cover_ring(ring, &mut win)? {
        let cov = win.coverage();
        for jr in 0..win.nrow() {
            for jc in 0..win.ncol() {
                let (r, c) = (win.row0() + jr as i64, win.col0() + jc as i64);
                if r >= 0 && c >= 0 && (r as usize) < m.nrow() && (c as usize) < m.ncol() {
                    out[r as usize * m.ncol() + c as usize] += cov[jr * win.ncol() + jc];
                }
            }
        }
    }
    Ok(out)
}

/// The world clip: each cell's covered fraction of its area, `None` for
/// cells with no area or an inexact split. The ring's orientation doesn't
/// matter.
fn oracle(m: &QuadMesh, ring: &[Pt]) -> Vec<Option<f64>> {
    let inexact = m.inexact_cells();
    let s = shoelace(ring).signum();
    let bx = |p: &[Pt]| {
        p.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, q| {
            [b[0].min(q[0]), b[1].min(q[1]), b[2].max(q[0]), b[3].max(q[1])]
        })
    };
    let rb = bx(ring);
    let mut out = vec![None; m.nrow() * m.ncol()];
    for r in 0..m.nrow() {
        for c in 0..m.ncol() {
            let a = m.cell_area(r, c);
            if a == 0.0 || inexact.contains(&(r, c)) {
                continue;
            }
            let q = m.corners(r, c);
            let cb = bx(&q);
            if cb[0] > rb[2] || cb[2] < rb[0] || cb[1] > rb[3] || cb[3] < rb[1] {
                out[r * m.ncol() + c] = Some(0.0);
                continue;
            }
            let x = m.split_centre(r, c).0;
            // the cell's area from its triangles, each from its own corner
            // (the mesh's own is a shoelace over absolute coordinates)
            let (mut cov, mut area) = (0.0, 0.0);
            for k in 0..4 {
                let mut tri = vec![x, q[k], q[(k + 1) % 4]];
                if shoelace(&tri) < 0.0 {
                    tri.reverse();
                }
                area += shoelace(&tri);
                let piece = clip(ring, &tri);
                if piece.len() >= 3 {
                    cov += s * shoelace(&piece);
                }
            }
            out[r * m.ncol() + c] = Some(cov / area);
        }
    }
    out
}

/// The cell where the walk and the clip differ most: `(row, col, walk,
/// clip)`.
fn worst(got: &[f64], want: &[Option<f64>], ncol: usize) -> Option<(usize, usize, f64, f64)> {
    let mut w: Option<(usize, usize, f64, f64)> = None;
    for (k, (g, o)) in got.iter().zip(want).enumerate() {
        if let Some(o) = o {
            let e = (g - o).abs();
            if w.is_none_or(|w| e > (w.2 - w.3).abs()) {
                w = Some((k / ncol, k % ncol, *g, *o));
            }
        }
    }
    w
}

#[derive(Default)]
struct Tally {
    cases: usize,
    failures: Vec<String>,
    /// Skip the ring reversed and turned (for the largest search).
    one_way: bool,
}

impl Tally {
    fn fail(&mut self, msg: String) {
        if std::env::var("WALK_SHOW").is_ok_and(|v| msg.starts_with(&v)) {
            eprintln!("{msg}");
        }
        self.failures.push(msg);
    }

    /// The walk against the world clip, and against the ring reversed and
    /// started at another vertex. Its coverage, unless the walk failed.
    fn ring(&mut self, name: &str, m: &QuadMesh, ring: &[Pt], tol: f64) -> Option<Vec<f64>> {
        self.cases += 1;
        let got = match cover(m, ring) {
            Ok(g) => g,
            Err(e) => {
                self.fail(format!("{name}: error {e}\n  ring {ring:?}"));
                return None;
            }
        };
        let want = oracle(m, ring);
        if let Some((r, c, g, o)) = worst(&got, &want, m.ncol()) {
            if (g - o).abs() > tol {
                self.fail(format!(
                    "{name}: cell ({r}, {c}) walk {g} clip {o} ({:.1e})\n  ring {ring:?}",
                    (g - o).abs()
                ));
                return Some(got);
            }
        }
        if self.one_way {
            return Some(got);
        }
        let mut rev = ring.to_vec();
        rev.reverse();
        // each segment's path is the same both ways, bit for bit
        if let (Ok(fwd), Ok(back)) = (m.ring_to_index(ring), m.ring_to_index(&rev)) {
            let mut back = back;
            back.reverse();
            let n = fwd.len();
            let same = n == back.len() && (n == 0 || (0..n).any(|s| (0..n).all(|i| fwd[(i + s) % n] == back[i])));
            if !same {
                self.fail(format!("{name}: reversed index ring isn't the index ring reversed\n  ring {ring:?}\n  fwd {fwd:?}\n  rev {back:?}"));
            }
        }
        let mut turned = ring.to_vec();
        turned.rotate_left(1 + ring.len() / 2);
        for (how, other) in [("reversed", rev), ("turned", turned)] {
            match cover(m, &other) {
                Ok(o) => {
                    let e = got.iter().zip(&o).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                    if e > SAME_TOL {
                        self.fail(format!("{name}: {how} differs by {e:.1e}\n  ring {ring:?}"));
                    }
                }
                Err(e) => self.fail(format!("{name}: {how}: error {e}\n  ring {ring:?}")),
            }
        }
        Some(got)
    }

    /// Pieces that tile `whole`: their coverages add up to its own.
    fn tiling(&mut self, name: &str, m: &QuadMesh, whole: &[Pt], pieces: &[Vec<Pt>], tol: f64) {
        let Some(want) = self.ring(&format!("{name} whole"), m, whole, tol) else {
            return;
        };
        let mut sum = vec![0.0; want.len()];
        for (i, p) in pieces.iter().enumerate() {
            let Some(got) = self.ring(&format!("{name} piece {i}"), m, p, tol) else {
                return;
            };
            for (s, g) in sum.iter_mut().zip(&got) {
                *s += g;
            }
        }
        self.cases += 1;
        let (k, e) = sum
            .iter()
            .zip(&want)
            .map(|(s, w)| (s - w).abs())
            .enumerate()
            .fold((0, 0.0), |b, (k, e)| if e > b.1 { (k, e) } else { b });
        if e > TILE_TOL {
            self.fail(format!(
                "{name}: pieces cover cell ({}, {}) {} but the whole {} ({e:.1e})\n  whole {whole:?}\n  pieces {pieces:?}",
                k / m.ncol(),
                k % m.ncol(),
                sum[k],
                want[k]
            ));
        }
    }

    /// Fails if any check did, with the failures counted by mesh and kind.
    fn check(self, what: &str) {
        eprintln!("{what}: {} checks, {} failed", self.cases, self.failures.len());
        let mut by: std::collections::BTreeMap<String, usize> = Default::default();
        for f in &self.failures {
            let mesh = f.split(' ').next().unwrap_or("");
            let kind = if f.contains(": error") {
                "error"
            } else if f.contains("reversed index ring") {
                "bitwise"
            } else if f.contains("reversed") {
                "reversed"
            } else if f.contains("turned") {
                "turned"
            } else if f.contains("pieces cover") {
                "tiling"
            } else {
                "clip"
            };
            *by.entry(format!("{mesh} {kind}")).or_default() += 1;
        }
        for (k, n) in &by {
            eprintln!("  {k}: {n}");
        }
        assert!(self.failures.is_empty(), "{what}: {} of {} checks failed", self.failures.len(), self.cases);
    }
}

/// More cases, for a longer search: `WALK_CASES=10 cargo test --release`.
fn cases(n: usize) -> usize {
    n * std::env::var("WALK_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

/// The world clip's tolerance: the walk's index fractions are exact to
/// rounding; the clip's area of a piece far from the origin is not.
const TOL: f64 = 1e-9;

/// Pieces' coverages against their union's: the same walk, so rounding.
const TILE_TOL: f64 = 1e-11;

/// A ring's coverage against the same ring reversed or turned.
const SAME_TOL: f64 = 1e-12;

// ---- generators -------------------------------------------------------------

/// The degenerate points near `c`, within `r`.
fn near(fs: &[Feature], c: Pt, r: f64) -> Vec<Feature> {
    fs.iter().filter(|f| norm(sub(f.p, c)) < r).copied().collect()
}

/// Rings whose vertices are degenerate points (or within rounding of
/// them), ordered by angle round a centre so the ring is simple.
#[test]
fn vertices_on_and_near_corners_sides_spokes_and_centres() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let mut g = Lcg(20261119);
        let fs = features(&m, &mut g);
        let h = scale(&m);
        for case in 0..cases(150) {
            let c = add(g.pick(&fs).p, [h * (g.u() - 0.5), h * (g.u() - 0.5)]);
            let r = h * (0.6 + 3.0 * g.u());
            let pool = near(&fs, c, r);
            if pool.len() < 3 {
                continue;
            }
            let n = 3 + g.below(8);
            let mut pts: Vec<(f64, Pt)> = (0..n)
                .map(|_| {
                    let q = g.pick(&pool).p;
                    let p = jiggle(&mut g, q, h);
                    let d = sub(p, c);
                    (d[1].atan2(d[0]), p)
                })
                .collect();
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            pts.dedup_by(|a, b| a.0 == b.0 || a.1 == b.1);
            // no gap of a half turn or more, so the ring is simple
            let gaps = (0..pts.len()).map(|i| {
                let a = pts[i].0;
                let b = pts[(i + 1) % pts.len()].0;
                if i + 1 == pts.len() { b + std::f64::consts::TAU - a } else { b - a }
            });
            if pts.len() < 3 || gaps.fold(0.0, f64::max) >= std::f64::consts::PI * 0.999 {
                continue;
            }
            let ring: Vec<Pt> = pts.iter().map(|p| p.1).collect();
            t.ring(&format!("{name} case {case}"), &m, &ring, TOL);
        }
    }
    t.check("vertices");
}

/// Convex polygons cut along lines through degenerate points: at random
/// angles, along a side or a spoke there, or through a second degenerate
/// point. The cut points are vertices of the pieces either side, and so
/// is the degenerate point when a piece's cut passes it.
#[test]
fn tilings_cut_through_corners_sides_spokes_and_centres() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let mut g = Lcg(47);
        let fs = features(&m, &mut g);
        let h = scale(&m);
        for case in 0..cases(60) {
            let c = g.pick(&fs).p;
            let whole = quad(c, h * (0.5 + 3.0 * g.u()), h * (0.5 + 3.0 * g.u()), g.u() * 3.0);
            let pool = near(&fs, c, 3.0 * h);
            if pool.is_empty() {
                continue;
            }
            let mut pieces = vec![whole.clone()];
            for _ in 0..1 + g.below(3) {
                let f = g.pick(&pool);
                let v = match g.below(3) {
                    0 => {
                        let a = g.u() * std::f64::consts::TAU;
                        [a.cos(), a.sin()]
                    }
                    1 => f.dir,
                    _ => sub(g.pick(&pool).p, f.p),
                };
                if norm(v) == 0.0 {
                    continue;
                }
                let d = jiggle(&mut g, f.p, h);
                pieces = pieces.iter().flat_map(|p| cut(p, d, v)).collect();
            }
            t.tiling(&format!("{name} case {case}"), &m, &whole, &pieces, TOL);
        }
    }
    t.check("cuts");
}

/// Triangles fanned round a degenerate point, some of their far corners
/// other degenerate points: they tile the star of the far corners.
#[test]
fn fans_round_corners_sides_spokes_and_centres() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let mut g = Lcg(5);
        let fs = features(&m, &mut g);
        let h = scale(&m);
        for case in 0..cases(80) {
            let f = g.pick(&fs);
            let d = jiggle(&mut g, f.p, h);
            let pool = near(&fs, d, 2.5 * h);
            let n = 3 + g.below(6);
            let mut ends: Vec<(f64, Pt)> = (0..n)
                .map(|i| {
                    let e = if g.u() < 0.5 {
                        let q = g.pick(&pool).p;
                        jiggle(&mut g, q, h)
                    } else {
                        let a = (i as f64 + 0.8 * g.u()) / n as f64 * std::f64::consts::TAU;
                        let r = h * (0.3 + 2.0 * g.u());
                        add(d, [r * a.cos(), r * a.sin()])
                    };
                    let v = sub(e, d);
                    (v[1].atan2(v[0]), e)
                })
                .collect();
            ends.sort_by(|a, b| a.0.total_cmp(&b.0));
            ends.dedup_by(|a, b| a.0 == b.0 || a.1 == b.1);
            ends.retain(|e| e.1 != d);
            let k = ends.len();
            if k < 3 {
                continue;
            }
            let gap = (0..k)
                .map(|i| {
                    let (a, b) = (ends[i].0, ends[(i + 1) % k].0);
                    if i + 1 == k { b + std::f64::consts::TAU - a } else { b - a }
                })
                .fold(0.0, f64::max);
            if gap >= std::f64::consts::PI * 0.999 {
                continue;
            }
            let star: Vec<Pt> = ends.iter().map(|e| e.1).collect();
            let wedges: Vec<Vec<Pt>> = (0..k).map(|i| vec![d, star[i], star[(i + 1) % k]]).collect();
            t.tiling(&format!("{name} case {case}"), &m, &star, &wedges, TOL);
        }
    }
    t.check("fans");
}

/// Rings along sides (a cell, a block of cells, with each corner a vertex
/// or only the block's own), and the same moved off them by a little.
#[test]
fn rings_along_and_beside_sides() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let mut g = Lcg(3);
        let h = scale(&m);
        for case in 0..cases(60) {
            let (r0, c0) = (g.below(m.nrow()), g.below(m.ncol()));
            let (r1, c1) = (r0 + g.below(m.nrow() - r0), c0 + g.below(m.ncol() - c0));
            let corner = |j: usize, i: usize| {
                if j < m.nrow() && i < m.ncol() {
                    m.corners(j, i)[0]
                } else if j < m.nrow() {
                    m.corners(j, i - 1)[1]
                } else if i < m.ncol() {
                    m.corners(j - 1, i)[3]
                } else {
                    m.corners(j - 1, i - 1)[2]
                }
            };
            let every = g.u() < 0.5;
            let mut ring = Vec::new();
            let mut push = |j: usize, i: usize, last: bool| {
                if every || last {
                    ring.push(corner(j, i));
                }
            };
            for i in c0..=c1 {
                push(r0, i, i == c0);
            }
            for j in r0..=r1 {
                push(j, c1 + 1, j == r0);
            }
            for i in (c0 + 1..=c1 + 1).rev() {
                push(r1 + 1, i, i == c1 + 1);
            }
            for j in (r0 + 1..=r1 + 1).rev() {
                push(j, c0, j == r1 + 1);
            }
            ring.dedup();
            if g.u() < 0.5 {
                // moved off the sides, out or in, by a little
                let d = h * g.pick(&[f64::EPSILON, 1e-14, 1e-12, 1e-10, -1e-14, -1e-12]);
                let cm = mul(ring.iter().fold([0.0, 0.0], |s, &p| add(s, p)), 1.0 / ring.len() as f64);
                ring = ring
                    .iter()
                    .map(|&p| {
                        let v = sub(p, cm);
                        add(p, mul(v, d / norm(v)))
                    })
                    .collect();
            }
            if ring.len() >= 3 && shoelace(&ring) != 0.0 {
                t.ring(&format!("{name} case {case}"), &m, &ring, TOL);
            }
        }
    }
    t.check("sides");
}

/// Thin quadrilaterals whose long side crosses a mesh side at a shallow
/// angle, or runs along it within rounding.
#[test]
fn shallow_crossings_of_sides() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let mut g = Lcg(11);
        let fs = features(&m, &mut g);
        let h = scale(&m);
        for case in 0..cases(120) {
            let f = g.pick(&fs);
            let v = f.dir;
            if norm(v) == 0.0 {
                continue;
            }
            let u = mul(v, 1.0 / norm(v));
            let n = [-u[1], u[0]];
            let tilt = h * g.pick(&[0.0, f64::EPSILON, 1e-15, 1e-13, 1e-11, 1e-8, 1e-5]);
            let (s0, s1) = (-0.3 - 2.0 * g.u(), 1.3 + 2.0 * g.u());
            let p = add(lerp(f.p, add(f.p, v), s0), mul(n, tilt));
            let q = sub(lerp(f.p, add(f.p, v), s1), mul(n, tilt));
            let w = mul(n, h * (0.1 + g.u()) * if g.u() < 0.5 { 1.0 } else { -1.0 });
            let ring = vec![p, q, add(q, w), add(p, w)];
            t.ring(&format!("{name} case {case}"), &m, &ring, TOL);
        }
    }
    t.check("shallow");
}

/// Rings through the mesh's own corners (a block of cells, each corner on
/// its outline a vertex, or on a mesh with straight lines of corners only
/// the block's four): exactly the block's cells covered, each exactly
/// once. Snapping puts a vertex on a corner, and a crossing within
/// rounding of one, exactly there, so no neighbour gets a sliver.
#[test]
fn rings_on_corners_cover_whole_cells_and_nothing_else() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let straight = ["square", "rotated", "rotated_far", "rotated_small", "flipped"].contains(&name);
        let mut g = Lcg(31);
        for case in 0..cases(60) {
            let (r0, c0) = (g.below(m.nrow()), g.below(m.ncol()));
            let (r1, c1) = (r0 + g.below(m.nrow() - r0), c0 + g.below(m.ncol() - c0));
            let corner = |j: usize, i: usize| {
                let (r, c) = (j.min(m.nrow() - 1), i.min(m.ncol() - 1));
                m.corners(r, c)[match (j > r, i > c) {
                    (false, false) => 0,
                    (false, true) => 1,
                    (true, true) => 2,
                    (true, false) => 3,
                }]
            };
            let every = !straight || g.u() < 0.5;
            let mut ring: Vec<Pt> = Vec::new();
            let lines: [(usize, usize, bool); 4] = [(r0, c0, true), (r0, c1 + 1, false), (r1 + 1, c1 + 1, true), (r1 + 1, c0, false)];
            for (s, &(j, i, along_row)) in lines.iter().enumerate() {
                let (nj, ni, _) = lines[(s + 1) % 4];
                ring.push(corner(j, i));
                if every {
                    if along_row {
                        let n = ni.abs_diff(i);
                        for step in 1..n {
                            ring.push(corner(j, if ni > i { i + step } else { i - step }));
                        }
                    } else {
                        let n = nj.abs_diff(j);
                        for step in 1..n {
                            ring.push(corner(if nj > j { j + step } else { j - step }, i));
                        }
                    }
                }
            }
            ring.dedup();
            if ring.len() < 3 {
                continue;
            }
            for (how, r) in [("", ring.clone()), (" reversed", ring.iter().rev().copied().collect())] {
                t.cases += 1;
                let got = match cover(&m, &r) {
                    Ok(v) => v,
                    Err(e) => {
                        t.fail(format!("{name} case {case}{how}: error {e}\n  ring {r:?}"));
                        continue;
                    }
                };
                for (k, v) in got.iter().enumerate() {
                    let (rr, cc) = (k / m.ncol(), k % m.ncol());
                    let inside = (r0..=r1).contains(&rr) && (c0..=c1).contains(&cc);
                    let want = if inside { 1.0 } else { 0.0 };
                    if *v != want {
                        t.fail(format!("{name} case {case}{how}: cell ({rr}, {cc}) covered {v:e}, not {want}\n  ring {r:?}"));
                        break;
                    }
                }
            }
        }
    }
    t.check("corners");
}

/// Fans round every corner of the mesh, the fan's centre the corner or a
/// rounding unit off it each way: the triangles tile the fan's star. A
/// point an ulp off a corner can be owned by a cell whose bounding box it
/// is outside, and the segments from it leave into the other cells round
/// the corner.
#[test]
fn fans_an_ulp_off_every_corner() {
    let mut t = Tally { one_way: true, ..Tally::default() };
    let nudge = |x: f64, k: i32| match k {
        -1 => x.next_down(),
        1 => x.next_up(),
        _ => x,
    };
    for (name, m) in meshes() {
        let h = scale(&m);
        let mut g = Lcg(1284);
        for j in 0..=m.nrow() {
            for i in 0..=m.ncol() {
                let (r, c) = (j.min(m.nrow() - 1), i.min(m.ncol() - 1));
                let k = match (j > r, i > c) {
                    (false, false) => 0,
                    (false, true) => 1,
                    (true, true) => 2,
                    (true, false) => 3,
                };
                let p = m.corners(r, c)[k];
                for (kx, ky) in (-1..=1).flat_map(|x| (-1..=1).map(move |y| (x, y))) {
                    let d = [nudge(p[0], kx), nudge(p[1], ky)];
                    let n = 5 + g.below(3);
                    let turn = g.u();
                    let star: Vec<Pt> = (0..n)
                        .map(|e| {
                            let a = (e as f64 + turn) / n as f64 * std::f64::consts::TAU;
                            let r = h * (0.4 + 0.8 * g.u());
                            add(d, [r * a.cos(), r * a.sin()])
                        })
                        .collect();
                    let wedges: Vec<Vec<Pt>> = (0..n).map(|e| vec![d, star[e], star[(e + 1) % n]]).collect();
                    t.tiling(&format!("{name} corner ({j}, {i}) nudged ({kx}, {ky})"), &m, &star, &wedges, TOL);
                }
            }
        }
    }
    t.check("ulp fans");
}

/// Long segments (1e5 cells) from inside a corner cell of the mesh that
/// leave it within `f` times 1e-12 of their length of the corner: the
/// last pieces in the mesh are shorter than the walk's joining tolerance
/// along the segment, and must still take it to the wall. Two triangles
/// share the segment, one either side; they tile a triangle without it.
#[test]
fn long_segments_leaving_past_an_outer_corner() {
    let mut t = Tally::default();
    for (name, m) in meshes() {
        let h = scale(&m);
        let (nr, nc) = (m.nrow(), m.ncol());
        let mut g = Lcg(77);
        for (r, c, k) in [(0, 0, 0), (0, nc - 1, 1), (nr - 1, nc - 1, 2), (nr - 1, 0, 3)] {
            let q = m.corners(r, c);
            let corner = q[k];
            let x = m.split_centre(r, c).0;
            for f in [0.0, 0.01, 0.1, 0.5, 0.9, -0.01, -0.1, -0.5, -0.9] {
                let inside = lerp(x, corner, 0.2 + 0.6 * g.u());
                let v = sub(corner, inside);
                let dv = norm(v);
                let u = mul(v, 1.0 / dv);
                let n = [-u[1], u[0]];
                let len = 1e5 * h;
                let graze = f * 1e-12 * len;
                let far = add(add(corner, mul(u, len)), mul(n, graze * (dv + len) / dv));
                let side = mul(n, 0.5 * h);
                let (r1, r2) = (add(inside, side), sub(inside, side));
                let pieces = vec![vec![inside, far, r1], vec![far, inside, r2]];
                t.tiling(&format!("{name} corner {k} graze {f}"), &m, &[r2, far, r1], &pieces, TOL);
            }
        }
    }
    t.check("long");
}
