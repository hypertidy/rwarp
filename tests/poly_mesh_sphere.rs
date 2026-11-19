//! Meshes with great-circle edges, ICON's and MPAS's kind, and their
//! conservative weights with a grid (conservative stage 5b,
//! `design/conservative.md` section 7): `PolyMesh::with_great_circles`.
//!
//! The meshes are made here: an icosahedron with nodes on the poles,
//! each face cut into triangles (a geodesic grid, as ICON's), and its
//! dual, a hexagon round each node and a pentagon round each of the
//! icosahedron's (MPAS's kind), two of them round the poles.
//!
//! Oracles: faces bounded by meridians and the equator against the
//! closed form of their overlaps; each face's area against the spherical
//! excess of its great-circle polygon, converging at bigcurve's rate;
//! the faces tiling a global grid, every cell covered once, across the
//! seam and round the poles; great circles cut into many chords here,
//! independently, as a planar mesh.

use std::cell::Cell;
use std::collections::HashMap;
use std::f64::consts::PI;

use rwarp::mesh::{PolyMesh, WORLD_GT};
use rwarp::transform::{CenterLong, GenImgProjTransformer};
use rwarp::weights::{lonlat_measure, GridWindow, WeightOptions, Weights};
use rwarp::{CrsTransform, Transformer};

struct Same;

impl CrsTransform for Same {
    fn convert(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        Some((x, y))
    }
}

/// From longitude and latitude to the pixels of a global grid of `d`
/// degree cells from -180, 90, its longitudes wrapped as GDAL's are.
fn global(d: f64) -> (GenImgProjTransformer, GridWindow, WeightOptions) {
    let gt = [-180.0, d, 0.0, 90.0, 0.0, -d];
    let (w, h) = ((360.0 / d) as usize, (180.0 / d) as usize);
    let t = GenImgProjTransformer::with_backend(gt, WORLD_GT, Box::new(Same), Box::new(Same))
        .unwrap()
        .with_center_long(CenterLong::At(0.0))
        .with_source_size([w, h]);
    (t, GridWindow::new(w, h), WeightOptions::new(1).with_measure(lonlat_measure(gt).unwrap()))
}

/// Steradians per pixel of a grid `d` degrees square, under the
/// spherical measure.
fn sr(d: f64) -> f64 {
    d.to_radians() * d.to_radians()
}

// ---- vectors and meshes on the sphere ------------------------------------------------

type V = [f64; 3];

fn add(a: V, b: V) -> V {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: V, s: f64) -> V {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V, b: V) -> V {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normal(a: V) -> V {
    scale(a, 1.0 / dot(a, a).sqrt())
}

fn vec(lon: f64, lat: f64) -> V {
    let (l, p) = (lon.to_radians(), lat.to_radians());
    [p.cos() * l.cos(), p.cos() * l.sin(), p.sin()]
}

/// Longitude and latitude of a unit vector; `pole_lon` on a pole.
fn lonlat(v: V, pole_lon: f64) -> (f64, f64) {
    let h = v[0].hypot(v[1]);
    if h < 1e-15 {
        return (pole_lon, 90f64.copysign(v[2]));
    }
    (v[1].atan2(v[0]).to_degrees(), v[2].atan2(h).to_degrees())
}

/// The area of the spherical polygon through `p` (great-circle edges,
/// convex), by a fan of triangles (Van Oosterom and Strackee).
fn excess(p: &[V]) -> f64 {
    let mut e = 0.0;
    for i in 1..p.len() - 1 {
        let (a, b, c) = (p[0], p[i], p[i + 1]);
        e += 2.0 * dot(a, cross(b, c)).atan2(1.0 + dot(a, b) + dot(b, c) + dot(c, a));
    }
    e.abs()
}

/// A mesh on the sphere: nodes as unit vectors, faces anticlockwise
/// from outside.
struct Sphere {
    v: Vec<V>,
    faces: Vec<Vec<usize>>,
}

impl Sphere {
    /// An icosahedron with nodes on both poles, each face cut into `n * n`
    /// triangles through points on the sphere.
    fn icosahedron(n: usize) -> Sphere {
        let lat = 0.5f64.atan().to_degrees();
        let mut corners = vec![vec(0.0, 90.0), vec(0.0, -90.0)];
        corners.extend((0..5).map(|k| vec(72.0 * k as f64, lat)));
        corners.extend((0..5).map(|k| vec(36.0 + 72.0 * k as f64, -lat)));
        let (u, l) = (|k: usize| 2 + k % 5, |k: usize| 7 + k % 5);
        let mut big = Vec::new();
        for k in 0..5 {
            big.push([0, u(k), u(k + 1)]);
            big.push([u(k), l(k), u(k + 1)]);
            big.push([u(k + 1), l(k), l(k + 1)]);
            big.push([1, l(k + 1), l(k)]);
        }
        let mut m = Sphere { v: Vec::new(), faces: Vec::new() };
        let mut index: HashMap<[i64; 3], usize> = HashMap::new();
        let mut node = |p: V, m: &mut Sphere| -> usize {
            let key = p.map(|c| (c * 1e9).round() as i64);
            *index.entry(key).or_insert_with(|| {
                m.v.push(p);
                m.v.len() - 1
            })
        };
        for [a, b, c] in big {
            let (a, b, c) = (corners[a], corners[b], corners[c]);
            let at = |i: usize, j: usize| {
                let (s, t) = (i as f64 / n as f64, j as f64 / n as f64);
                normal(add(a, add(scale(add(b, scale(a, -1.0)), s), scale(add(c, scale(a, -1.0)), t))))
            };
            for i in 0..n {
                for j in 0..n - i {
                    let tri = [at(i, j), at(i + 1, j), at(i, j + 1)];
                    let f = tri.map(|p| node(p, &mut m));
                    m.push(f.to_vec());
                    if i + j + 1 < n {
                        let tri = [at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                        let f = tri.map(|p| node(p, &mut m));
                        m.push(f.to_vec());
                    }
                }
            }
        }
        m
    }

    /// A face, turned anticlockwise from outside if it isn't.
    fn push(&mut self, mut f: Vec<usize>) {
        let (a, b, c) = (self.v[f[0]], self.v[f[1]], self.v[f[2]]);
        if dot(a, cross(add(b, scale(a, -1.0)), add(c, scale(a, -1.0)))) < 0.0 {
            f.reverse();
        }
        self.faces.push(f);
    }

    /// The dual: a node at each face's centre (on the sphere), and a face
    /// round each node through the centres of the faces about it.
    fn dual(&self) -> Sphere {
        let v: Vec<V> = self.faces.iter().map(|f| normal(f.iter().fold([0.0; 3], |s, &k| add(s, self.v[k])))).collect();
        let mut about = vec![Vec::new(); self.v.len()];
        for (k, f) in self.faces.iter().enumerate() {
            for &n in f {
                about[n].push(k);
            }
        }
        let mut faces = Vec::new();
        for (n, ts) in about.into_iter().enumerate() {
            let p = self.v[n];
            let e1 = normal(cross(if p[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }, p));
            let e2 = cross(p, e1);
            let mut ts: Vec<(f64, usize)> = ts.into_iter().map(|t| (dot(v[t], e2).atan2(dot(v[t], e1)), t)).collect();
            ts.sort_by(|a, b| a.0.total_cmp(&b.0));
            faces.push(ts.into_iter().map(|t| t.1).collect());
        }
        let mut d = Sphere { v, faces: Vec::new() };
        for f in faces {
            d.push(f);
        }
        d
    }

    /// As a `PolyMesh` with great-circle edges, a pole's node written at
    /// `pole_lon`.
    fn mesh(&self, pole_lon: f64) -> PolyMesh {
        let (x, y): (Vec<f64>, Vec<f64>) = self.v.iter().map(|&p| lonlat(p, pole_lon)).unzip();
        let mut start = vec![0];
        let mut nodes = Vec::new();
        for f in &self.faces {
            nodes.extend(f);
            start.push(nodes.len());
        }
        PolyMesh::new(x, y, start, nodes).unwrap().with_great_circles().unwrap()
    }

    fn excess(&self, f: usize) -> f64 {
        excess(&self.faces[f].iter().map(|&k| self.v[k]).collect::<Vec<_>>())
    }
}

/// Per grid cell, the sum of onto's `frac` over the faces.
fn frac_per_cell(w: &Weights, ncol: usize, ncell: usize) -> Vec<f64> {
    let mut s = vec![0.0; ncell];
    for k in 0..w.len() {
        s[w.src_row[k] as usize * ncol + w.src_col[k] as usize] += w.frac[k];
    }
    s
}

fn worst(v: impl IntoIterator<Item = f64>) -> f64 {
    v.into_iter().fold(0.0, f64::max)
}

/// Counts the points it moves.
struct Counting<'a, T> {
    inner: T,
    n: &'a Cell<usize>,
}

impl<T: Transformer> Transformer for Counting<'_, T> {
    fn transform(&self, d: bool, x: &mut [f64], y: &mut [f64]) -> Vec<bool> {
        self.n.set(self.n.get() + x.len());
        self.inner.transform(d, x, y)
    }
    fn source_wrap(&self) -> Option<rwarp::transform::SourceWrap> {
        self.inner.source_wrap()
    }
}

// ---- the tests -------------------------------------------------------------------------

/// Triangles from the equator to a pole are lon/lat boxes: their edges are
/// two meridians and the equator, all great circles. On a 10 degree grid
/// every overlap is a whole cell, at any densify, refined or not, wherever
/// the pole's node is written.
#[test]
fn meridians_and_the_equator_are_exact() {
    let (mut x, mut y) = (vec![33.0, -71.0], vec![90.0, -90.0]);
    for k in 0..12 {
        x.push(-180.0 + 30.0 * k as f64);
        y.push(0.0);
    }
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for k in 0..12 {
        let (a, b) = (2 + k, 2 + (k + 1) % 12);
        nodes.extend([a, b, 0, b, a, 1]);
        start.extend([nodes.len() - 3, nodes.len()]);
    }
    let mesh = PolyMesh::new(x, y, start, nodes).unwrap().with_great_circles().unwrap();
    let (t, grid, opts) = global(10.0);
    for opts in [opts, opts.with_densify(5), opts.with_refine_tolerance(1e-4)] {
        let onto = mesh.weights_onto(&t, &grid, &opts).unwrap();
        assert_eq!(onto.len(), 24 * 27, "{opts:?}");
        let d = worst(onto.frac.iter().map(|f| (f - 1.0).abs()));
        let a = worst(onto.dst_area.iter().map(|a| (a * sr(10.0) - PI / 6.0).abs()));
        assert!(d < 1e-13 && a < 1e-14, "{opts:?}: {d:e} {a:e}");
        for k in 0..onto.len() {
            // face 2k north of the equator from -180 + 30k, 2k + 1 south
            let (f, r, c) = (onto.dst_col[k] as usize, onto.src_row[k] as usize, onto.src_col[k] as usize);
            assert_eq!((c / 3, r / 9), (f / 2, f % 2), "face {f} on cell ({r}, {c})");
        }
    }
}

/// How far a mesh's faces are from their spherical excess, in pixels,
/// refined to each tolerance, and the worst of `|frac sum - 1|` over the
/// cells of the global grid `d` degrees square.
fn converge(sph: &Sphere, mesh: &PolyMesh, d: f64, tols: &[f64], n: &Cell<usize>) -> Vec<(f64, f64, f64)> {
    let (t, grid, opts) = global(d);
    let t = Counting { inner: t, n };
    let ncol = grid.ncol;
    tols.iter()
        .map(|&tol| {
            n.set(0);
            let onto = mesh.weights_onto(&t, &grid, &opts.with_refine_tolerance(tol)).unwrap();
            let err = worst((0..mesh.nface()).map(|f| (onto.dst_area[f] - sph.excess(f) / sr(d)).abs()));
            let tiles = worst(frac_per_cell(&onto, ncol, ncol * grid.nrow).into_iter().map(|s| (s - 1.0).abs()));
            let total: f64 = onto.dst_area.iter().sum::<f64>() * sr(d);
            eprintln!(
                "tolerance {tol:e}: worst face {err:.2e} pixels ({:.1} tolerances), cells to {tiles:.1e}, total - 4 pi {:.1e}, {} points",
                err / tol,
                total - 4.0 * PI,
                n.get()
            );
            (err, tiles, n.get() as f64)
        })
        .collect()
}

/// The error follows the tolerance: within a factor of 4 of one ratio to
/// it, and down by 4 or more a decade.
fn follows(tols: &[f64], r: &[(f64, f64, f64)]) {
    let ratio: Vec<f64> = tols.iter().zip(r).map(|(t, e)| e.0 / t).collect();
    let (lo, hi) = (ratio.iter().copied().fold(f64::INFINITY, f64::min), worst(ratio.iter().copied()));
    assert!(hi < 4.0 * lo, "{ratio:?}");
    assert!(r.windows(2).all(|w| w[1].0 < w[0].0 / 4.0), "{r:?}");
}

/// The geodesic mesh (320 triangles) onto a global 2 degree grid: each
/// face's area converges on its spherical excess at the tolerance's rate,
/// the points as its inverse square root, and the faces tile the grid at
/// every tolerance (every cell's `frac` sums to 1).
#[test]
fn geodesic_faces_converge_on_their_spherical_excess() {
    let ico = Sphere::icosahedron(4);
    let mesh = ico.mesh(0.0);
    assert_eq!(mesh.nface(), 320);
    let tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5];
    let n = Cell::new(0);
    let r = converge(&ico, &mesh, 2.0, &tols, &n);
    assert!(r.iter().all(|e| e.1 < 1e-12), "{r:?}");
    follows(&tols, &r);
    let rate = (r[4].2 / r[1].2).log10() / 3.0;
    eprintln!("points per decade of tolerance from 1e-2: 10^{rate:.3}");
    assert!((0.45..0.55).contains(&rate), "{rate}");
}

/// Without refinement, `densify` points on each great circle: the faces
/// converge on their excess as its inverse square, and tile the grid.
#[test]
fn a_fixed_densify_follows_the_great_circles() {
    let ico = Sphere::icosahedron(4);
    let mesh = ico.mesh(0.0);
    let (t, grid, opts) = global(2.0);
    let mut errs = Vec::new();
    for d in [8, 32, 128] {
        let onto = mesh.weights_onto(&t, &grid, &opts.with_densify(d)).unwrap();
        let err = worst((0..320).map(|f| (onto.dst_area[f] - ico.excess(f) / sr(2.0)).abs()));
        let tiles = worst(frac_per_cell(&onto, 180, 16200).into_iter().map(|s| (s - 1.0).abs()));
        eprintln!("densify {d}: worst face {err:.2e} pixels, cells to {tiles:.1e}");
        assert!(tiles < 1e-12, "densify {d}: {tiles:e}");
        errs.push(err);
    }
    let rate = [(errs[0] / errs[1]).log(4.0), (errs[1] / errs[2]).log(4.0)];
    eprintln!("error against densify: densify^-{:.2}, ^-{:.2}", rate[0], rate[1]);
    assert!(rate.iter().all(|r| (1.8..2.2).contains(r)), "{rate:?}");
}

/// The dual mesh, hexagons and twelve pentagons, two of them round the
/// poles: the faces tile the grid and each is its excess, the pole's too.
#[test]
fn cells_round_the_poles() {
    let dual = Sphere::icosahedron(4).dual();
    let mesh = dual.mesh(0.0);
    assert_eq!(mesh.nface(), 162);
    let poles: Vec<usize> = (0..162).filter(|&f| (0..mesh.face(f).len()).all(|i| mesh.node(mesh.face(f)[i])[1].abs() > 80.0)).collect();
    assert_eq!(poles.len(), 2);
    let tols = [1e-2, 1e-3, 1e-4, 1e-5];
    let r = converge(&dual, &mesh, 2.0, &tols, &Cell::new(0));
    assert!(r.iter().all(|e| e.1 < 1e-12), "{r:?}");
    follows(&tols, &r);
    // a pole's pentagon covers its cap of cells wholly
    let (t, grid, opts) = global(2.0);
    let onto = mesh.weights_onto(&t, &grid, &opts.with_refine_tolerance(1e-4)).unwrap();
    for &f in &poles {
        let whole = (0..onto.len()).filter(|&k| onto.dst_col[k] as usize == f && onto.frac[k] > 1.0 - 1e-12);
        assert_eq!(whole.filter(|&k| onto.src_row[k] == 0 || onto.src_row[k] == 89).count(), 180, "face {f}");
    }
}

/// Where the pole's node is written doesn't matter.
#[test]
fn a_pole_s_longitude_doesnt_matter() {
    let ico = Sphere::icosahedron(2);
    let (t, grid, opts) = global(5.0);
    for opts in [opts.with_densify(4), opts.with_refine_tolerance(1e-4)] {
        let a = ico.mesh(0.0).weights_onto(&t, &grid, &opts).unwrap();
        let b = ico.mesh(123.0).weights_onto(&t, &grid, &opts).unwrap();
        let key = |w: &Weights, k: usize| (w.dst_col[k], w.src_row[k], w.src_col[k]);
        let ea: HashMap<_, _> = (0..a.len()).map(|k| (key(&a, k), a.frac[k])).collect();
        let d = worst((0..b.len()).map(|k| (b.frac[k] - ea.get(&key(&b, k)).copied().unwrap_or(0.0)).abs()));
        assert!(d < 1e-12 && ea.len() == b.len(), "{opts:?}: {d:e}");
    }
}

/// Great circles cut into 2048 chords each, here, by hand, as a planar mesh
/// in longitude and latitude, give what the great-circle mesh gives,
/// refined: on the faces away from the poles (where a planar mesh's node
/// on a pole is taken as written).
#[test]
fn great_circles_are_their_many_chords() {
    let ico = Sphere::icosahedron(2);
    let keep: Vec<usize> = (0..ico.faces.len()).filter(|&f| ico.faces[f].iter().all(|&k| ico.v[k][2].abs() < 0.95)).collect();
    let gc = {
        let m = ico.mesh(0.0);
        let (x, y): (Vec<f64>, Vec<f64>) = (0..m.nnode()).map(|n| (m.node(n)[0], m.node(n)[1])).unzip();
        let mut start = vec![0];
        let mut nodes = Vec::new();
        for &f in &keep {
            nodes.extend(m.face(f));
            start.push(nodes.len());
        }
        PolyMesh::new(x, y, start, nodes).unwrap().with_great_circles().unwrap()
    };
    // the chords: each edge's 2047 interior points made once, from its
    // smaller node, so faces share them
    let (mut x, mut y): (Vec<f64>, Vec<f64>) = ico.v.iter().map(|&p| lonlat(p, 0.0)).unzip();
    let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let (mut start, mut nodes) = (vec![0], Vec::new());
    for &f in &keep {
        let v = &ico.faces[f];
        for i in 0..v.len() {
            let (a, b) = (v[i], v[(i + 1) % v.len()]);
            let inner = edges.entry((a.min(b), a.max(b))).or_insert_with(|| {
                let (p, q) = (ico.v[a.min(b)], ico.v[a.max(b)]);
                let w = dot(p, q).acos();
                (1..2048)
                    .map(|k| {
                        let t = k as f64 / 2048.0;
                        let r = add(scale(p, ((1.0 - t) * w).sin() / w.sin()), scale(q, (t * w).sin() / w.sin()));
                        let (lon, lat) = lonlat(r, 0.0);
                        x.push(lon);
                        y.push(lat);
                        x.len() - 1
                    })
                    .collect()
            });
            nodes.push(a);
            if a < b {
                nodes.extend(inner.iter());
            } else {
                nodes.extend(inner.iter().rev());
            }
        }
        start.push(nodes.len());
    }
    let chords = PolyMesh::new(x, y, start, nodes).unwrap().with_period(360.0).unwrap();
    let (t, grid, opts) = global(5.0);
    let a = gc.weights_onto(&t, &grid, &opts.with_refine_tolerance(1e-7)).unwrap();
    let b = chords.weights_onto(&t, &grid, &opts).unwrap();
    let key = |w: &Weights, k: usize| (w.dst_col[k], w.src_row[k], w.src_col[k]);
    let eb: HashMap<_, _> = (0..b.len()).map(|k| (key(&b, k), b.frac[k])).collect();
    let d = worst((0..a.len()).map(|k| (a.frac[k] - eb.get(&key(&a, k)).copied().unwrap_or(0.0)).abs()));
    let area = worst(a.dst_area.iter().zip(&b.dst_area).map(|(p, q)| (p - q).abs()));
    eprintln!("{} faces: overlaps to {d:.1e}, areas to {area:.1e} pixels", keep.len());
    assert!(d < 1e-6 && area < 1e-6, "{d:e} {area:e}");
}

/// Refusals: a latitude beyond a pole, an edge between antipodes, a period
/// other than 360.
#[test]
fn refusals() {
    let tri = |y2: f64, x1: f64| PolyMesh::new(vec![0.0, x1, 10.0], vec![0.0, 0.0, y2], vec![0, 3], vec![0, 1, 2]).unwrap();
    assert!(tri(91.0, 5.0).with_great_circles().unwrap_err().contains("beyond a pole"));
    let anti = PolyMesh::new(vec![0.0, 180.0, 90.0], vec![0.0, 0.0, 45.0], vec![0, 3], vec![0, 1, 2]).unwrap();
    assert!(anti.with_great_circles().unwrap_err().contains("antipodes"));
    let m = tri(10.0, 5.0).with_great_circles().unwrap();
    assert!(m.great_circles() && m.period() == Some(360.0));
    assert!(m.clone().with_period(180.0).is_err());
    assert!(m.with_period(360.0).is_ok());
}

/// In longitude and latitude a great circle across the equator is an S:
/// its curve turns there. An edge that crosses it at its middle has its
/// middle on its chord, where refinement's test can't see the curve, so
/// it is cut at the equator first. Refined at densify 1, edges from 20 to
/// 62 degrees long crossing at their middle or a quarter of the way give
/// what densify 2048 gives.
#[test]
fn an_arc_across_the_equator_is_followed() {
    let (t, grid, opts) = global(2.0);
    let key = |w: &Weights, k: usize| (w.src_row[k], w.src_col[k]);
    for h in [8.0f64, 12.5, 25.0] {
        for far in [1.0, 3.0] {
            // the long edge from (-h, -0.75 h) to (far h, 0.75 far h)
            let (x, y) = (vec![-h, far * h, -h], vec![-0.75 * h, 0.75 * far * h, 0.75 * far * h]);
            let m = PolyMesh::new(x, y, vec![0, 3], vec![0, 1, 2]).unwrap().with_great_circles().unwrap();
            let fine = m.weights_onto(&t, &grid, &opts.with_densify(2048)).unwrap();
            let ef: HashMap<_, _> = (0..fine.len()).map(|k| (key(&fine, k), fine.frac[k])).collect();
            let w = m.weights_onto(&t, &grid, &opts.with_refine_tolerance(1e-6)).unwrap();
            let d = worst((0..w.len()).map(|k| (w.frac[k] - ef.get(&key(&w, k)).copied().unwrap_or(0.0)).abs()));
            assert!(d < 2e-6, "h {h}, far {far}: {d:e}");
        }
    }
}

/// Every value of a table, bit for bit.
fn bits(w: &Weights) -> Vec<u64> {
    let mut v: Vec<u64> = [w.dst_off, w.dst_size].iter().flatten().map(|&x| x as u64).collect();
    for ix in [&w.dst_row, &w.dst_col, &w.src_row, &w.src_col] {
        v.push(ix.len() as u64);
        v.extend(ix.iter().map(|&x| x as u64));
    }
    for f in [&w.frac, &w.dst_area, &w.src_cell_area] {
        v.push(f.len() as u64);
        v.extend(f.iter().map(|x| x.to_bits()));
    }
    v
}

/// On threads, the serial table bit for bit, great circles refined: the
/// geodesic mesh and its dual, both ways.
#[test]
fn threads_are_the_serial_table() {
    let ico = Sphere::icosahedron(4);
    let meshes = [ico.mesh(0.0), ico.dual().mesh(0.0)];
    let make = || Ok::<_, String>(global(2.0).0);
    let (t, grid, opts) = global(2.0);
    let opts = opts.with_refine_tolerance(1e-3);
    for m in &meshes {
        let (onto, to) = (bits(&m.weights_onto(&t, &grid, &opts).unwrap()), bits(&m.weights_to(&t, &grid, &opts).unwrap()));
        for threads in [2, 5] {
            assert!(bits(&m.weights_onto_par(make, &grid, &opts, threads).unwrap()) == onto, "{threads} threads: onto");
            assert!(bits(&m.weights_to_par(make, &grid, &opts, threads).unwrap()) == to, "{threads} threads: to");
        }
    }
}
